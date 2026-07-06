"""Phase 1 — Graph-Guided Trajectory Distillation.

Fixes bug #4: the legacy code's ``get_retrieved_text_placeholder`` always
returned a literal placeholder string. The Teacher now sees the actual
RRF top-K passages retrieved through :mod:`kgproweight.retrieval.hybrid`.

Entry-point: :func:`run_phase1`.
"""

from __future__ import annotations

import json
import os
import string
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import re

from kgproweight.data.parsers import (
    extract_final_answer,
    parse_steps,
)
from kgproweight.data.prompts import build_teacher_messages
from kgproweight.data.silver_dataset import SilverDatasetReader, SilverStepRecord, SilverTrajectory
from kgproweight.kg.coverage import coverage_score
from kgproweight.kg.entity_linker import EntityLinker, extract_mentions
from kgproweight.kg.wikidata_retriever import WikidataSubgraphRetriever
from kgproweight.reward.prm_annotator import PRMAnnotator
from kgproweight.retrieval.hybrid import DEFAULT_TOPK
from kgproweight.utils.logging import dump_manifest, get_logger
from kgproweight.utils.paths import data_dir, index_dir

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _token_f1(pred: str, gold: str) -> float:
    """Standard token-level F1, normalised."""
    import re
    import string

    def norm(s: str) -> str:
        s = s.lower().strip()
        s = re.sub(r"\b(a|an|the)\b", " ", s)
        s = s.translate(str.maketrans("", "", string.punctuation))
        return " ".join(s.split())

    p = norm(pred).split()
    g = norm(gold).split()
    if not p or not g:
        return 0.0
    common = set(p) & set(g)
    if not common:
        return 0.0
    n_same = sum(min(p.count(t), g.count(t)) for t in common)
    if n_same == 0:
        return 0.0
    prec = n_same / len(p)
    rec = n_same / len(g)
    return 2 * prec * rec / (prec + rec)


# ---------------------------------------------------------------------------
# Lenient answer matching (ported verbatim from the validated `try` variant)
# ---------------------------------------------------------------------------

_ARTICLE_RE = re.compile(r"\b(a|an|the)\b")
_LEADIN_RE = re.compile(
    r"^(the\s+answer\s+is|answer\s*:|final\s+answer\s*:|it\s+is|this\s+is)\s+",
    re.IGNORECASE,
)


def normalize_answer(s: str) -> str:
    """SQuAD-style normalisation: lower, strip articles, strip punctuation."""
    s = s.lower().strip()
    s = _ARTICLE_RE.sub(" ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


def clean_final_answer(pred: str) -> str:
    """Reduce a possibly-verbose final answer to its core entity phrase.

    - drop a leading "the answer is" / "answer:" lead-in,
    - keep only the first line,
    - keep only the part before the first sentence-final separator so that
      "Albert Einstein, a physicist" → "Albert Einstein".
    """
    if not pred:
        return ""
    text = pred.strip().splitlines()[0].strip()
    text = _LEADIN_RE.sub("", text).strip()
    # Cut at the first comma / semicolon / dash / "(" / " because"/" which".
    text = re.split(r"[,;(]| because | which | that | was | is ", text, maxsplit=1)[0]
    return text.strip().strip(".").strip()


def _token_recall(pred: str, gold: str) -> float:
    """Fraction of gold tokens present in pred — robust to verbose preds."""
    p = set(normalize_answer(pred).split())
    g = normalize_answer(gold).split()
    if not g:
        return 0.0
    hit = sum(1 for t in g if t in p)
    return hit / len(g)


def answer_match_score(pred: str, gold: str) -> float:
    """Lenient match score in ``[0, 1]`` used for silver acceptance.

    The score is the maximum over three views so that a *correct* answer
    surrounded by extra words is not penalised by precision:

      * exact match after normalisation                       → 1.0
      * gold is a contiguous substring of the cleaned pred    → 1.0
      * token recall of gold inside pred (handles verbosity)
      * plain token-F1 on the cleaned pred (handles aliases)

    ``clean_final_answer`` is applied first to strip lead-ins / trailing
    clauses, which is where the original strict F1 lost most good traces.
    """
    if not pred or not gold:
        return 0.0
    cleaned = clean_final_answer(pred)
    n_pred_full = normalize_answer(pred)
    n_pred_clean = normalize_answer(cleaned)
    n_gold = normalize_answer(gold)
    if not n_gold:
        return 0.0

    # Exact (either on the cleaned or full pred).
    if n_gold == n_pred_clean or n_gold == n_pred_full:
        return 1.0
    # Substring: gold fully contained as a token sequence in pred.
    if n_gold and (f" {n_gold} " in f" {n_pred_full} " or f" {n_gold} " in f" {n_pred_clean} "):
        return 1.0

    recall = max(_token_recall(cleaned, gold), _token_recall(pred, gold))
    f1 = max(_token_f1(cleaned, gold), _token_f1(pred, gold))
    return max(recall, f1)


# ---------------------------------------------------------------------------
# Robust mention extraction (passage-title anchors + optional spaCy NER)
# Ported verbatim from the validated `try` variant.
# ---------------------------------------------------------------------------

_CAP_PHRASE_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")
_MENTION_BLACKLIST = {
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "is", "are", "was", "were", "do", "does", "did", "can", "could",
    "should", "would", "will", "the", "a", "an",
}

# Lazily-loaded spaCy pipeline (None means "not attempted yet", False means
# "tried and unavailable").
_SPACY_NLP: Any = None


def _maybe_spacy():
    global _SPACY_NLP
    if _SPACY_NLP is not None:
        return _SPACY_NLP or None
    try:
        import spacy  # type: ignore

        _SPACY_NLP = spacy.load("en_core_web_sm", disable=["lemmatizer", "tagger"])
    except Exception:
        _SPACY_NLP = False
    return _SPACY_NLP or None


def _passage_title(p: Any) -> Optional[str]:
    """Pull the title out of a FlashRAG passage dict (``contents = title\\ntext``)."""
    if isinstance(p, dict):
        title = p.get("title")
        if title:
            return str(title).strip()
        contents = str(p.get("contents") or "").strip()
        if contents:
            return contents.splitlines()[0].strip()
    return None


def extract_mentions_robust(
    question: str,
    passages: Optional[Sequence[Any]] = None,
    max_n: int = 8,
    title_anchor_top: int = 5,
) -> List[str]:
    """Best-effort mention set combining three sources.

    Order of preference (deduplicated, capped at ``max_n``):
      1. spaCy NER entities on the question (if spaCy is installed),
      2. capitalised noun phrases from the question (regex),
      3. titles of the top retrieved passages (strong anchors — the gold
         supporting docs in HotpotQA/2Wiki are titled by their key entity).
    """
    seen: Dict[str, None] = {}

    def _add(m: str) -> None:
        m = m.strip()
        if len(m) < 3:
            return
        if m.lower() in _MENTION_BLACKLIST:
            return
        seen.setdefault(m, None)

    nlp = _maybe_spacy()
    if nlp is not None and question:
        try:
            for ent in nlp(question).ents:
                if ent.label_ in {"PERSON", "ORG", "GPE", "LOC", "FAC", "EVENT", "WORK_OF_ART", "NORP", "PRODUCT"}:
                    _add(ent.text)
        except Exception:
            pass

    for m in _CAP_PHRASE_RE.findall(question or ""):
        _add(m)

    if passages:
        for p in list(passages)[:title_anchor_top]:
            title = _passage_title(p)
            if title:
                _add(title)

    return list(seen.keys())[:max_n]


def _select_relevant_triples(
    question: str,
    passages: Sequence[Dict[str, Any]],
    triples: Sequence[Sequence[str]],
    top_n: int,
) -> List[Sequence[str]]:
    """Keep KG context focused so the teacher can cite it more reliably.

    Scores each triple by token overlap with question + top retrieved passages.
    Falls back to original order when overlap is not informative.
    """
    if not triples:
        return []
    if len(triples) <= top_n:
        return list(triples)

    text_blobs: List[str] = [question]
    for p in list(passages)[:8]:
        t = str(p.get("contents") or p.get("text") or "").strip()
        if t:
            text_blobs.append(t[:1200])
    context = " ".join(text_blobs).lower()
    ctx_tokens = set(re.findall(r"[a-z0-9]+", context))
    if not ctx_tokens:
        return list(triples)[:top_n]

    scored: List[tuple[int, int, Sequence[str]]] = []
    for i, tri in enumerate(triples):
        if len(tri) != 3:
            continue
        tri_text = " ".join(str(x) for x in tri).lower()
        tri_tokens = set(re.findall(r"[a-z0-9]+", tri_text))
        overlap = len(ctx_tokens & tri_tokens)
        scored.append((overlap, -i, tri))

    scored.sort(reverse=True)
    picked = [t for _, _, t in scored[:top_n]]
    if not any(s > 0 for s, _, _ in scored[:top_n]):
        return list(triples)[:top_n]
    return picked


# ---------------------------------------------------------------------------
# Teacher client
# ---------------------------------------------------------------------------

@dataclass
class TeacherClient:
    """Tiny wrapper around an OpenAI-compatible chat client."""

    model: str = "deepseek-chat"
    backend: str = "deepseek"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.3
    max_tokens: int = 1500
    max_retries: int = 3

    def __post_init__(self) -> None:
        from openai import OpenAI

        if self.backend == "deepseek":
            api_key = self.api_key or os.environ.get("DEEPSEEK_API_KEY")
            base_url = self.base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        else:
            api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
            base_url = self.base_url or os.environ.get("OPENAI_BASE_URL")
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def chat(self, messages: List[Dict[str, str]]) -> str:
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning("Teacher attempt %d/%d failed: %s", attempt + 1, self.max_retries, exc)
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"Teacher generation failed after {self.max_retries} attempts: {last_exc}")


# ---------------------------------------------------------------------------
# Retrieval adapter
# ---------------------------------------------------------------------------

class _RetrievalAdapter:
    """Minimal adapter around FlashRAG's retriever for Phase 1.

    A real call to the FlashRAG retriever is heavy (loads e5 + FAISS); we
    therefore allow callers to pass in any object with a ``search(query)``
    method that returns a list of ``{"contents": str}`` dicts. This is
    used both in unit tests and in :mod:`scripts.train.phase1_generate_silver`.
    """

    def __init__(self, retriever: Any, top_k: int = DEFAULT_TOPK) -> None:
        self.retriever = retriever
        self.top_k = top_k

    def __call__(self, query: str) -> List[Dict[str, Any]]:
        if hasattr(self.retriever, "search"):
            results = self.retriever.search(query)
        elif hasattr(self.retriever, "batch_search"):
            results = self.retriever.batch_search([query])[0]
        else:
            return []
        return list(results)[: self.top_k]


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

@dataclass
class SilverFilter:
    min_steps: int = 3
    max_steps: int = 7
    min_triple_rate: float = 0.4
    min_coverage: float = 0.5
    min_token_f1: float = 0.5

    def accepts(
        self,
        steps,
        coverage: float,
        token_f1: float,
    ) -> bool:
        n = len(steps)
        if n < self.min_steps or n > self.max_steps:
            return False
        if coverage < self.min_coverage:
            return False
        if token_f1 < self.min_token_f1:
            return False
        n_with_triples = sum(1 for s in steps if s.cited_triples)
        if n_with_triples / max(n, 1) < self.min_triple_rate:
            return False
        return True


# ---------------------------------------------------------------------------
# Stratified acceptance filter (ported verbatim from the validated `try`
# variant). This is the new default; the hard ``SilverFilter`` above is kept
# for back-compat.
# ---------------------------------------------------------------------------

@dataclass
class StratifiedDecision:
    accepted: bool
    bucket: str          # "kg_rich" | "kg_medium" | "kg_sparse" | "rejected_quality"
    triple_rate: float
    reason: str


@dataclass
class StratifiedSilverFilter:
    """Accept trajectories by KG-density bucket with per-bucket quotas.

    Unlike the original hard ``triple_rate``/``coverage`` rejection, low-KG
    trajectories are *not* discarded outright. They are routed to a
    ``kg_sparse`` bucket which keeps up to ``sparse_quota`` of the total
    accepted count, so the α-Gate sees genuine α→0 fallback examples.

    Quality gates that everyone must pass (regardless of bucket):
      * step count within ``[min_steps, max_steps]``,
      * lenient answer-match score ≥ ``min_answer_score``.

    ``coverage`` and ``triple_rate`` are recorded but never hard-reject; they
    only decide the bucket and are subject to quota.
    """

    min_steps: int = 3
    max_steps: int = 7
    min_answer_score: float = 0.3
    # bucket thresholds on triple_rate
    rich_triple_rate: float = 0.5
    medium_triple_rate: float = 0.15
    # fraction of the accepted pool allowed to come from the sparse bucket
    sparse_quota: float = 0.25
    medium_quota: float = 0.35

    # running counters (mutated as trajectories stream in)
    _counts: Dict[str, int] = field(default_factory=lambda: {"kg_rich": 0, "kg_medium": 0, "kg_sparse": 0})

    @property
    def total_accepted(self) -> int:
        return sum(self._counts.values())

    def _bucket_for(self, triple_rate: float) -> str:
        if triple_rate >= self.rich_triple_rate:
            return "kg_rich"
        if triple_rate >= self.medium_triple_rate:
            return "kg_medium"
        return "kg_sparse"

    def decide(self, steps, coverage: float, answer_score: float) -> StratifiedDecision:
        n = len(steps)
        n_with_triples = sum(1 for s in steps if s.cited_triples)
        triple_rate = n_with_triples / max(n, 1)

        # universal quality gates
        if n < self.min_steps or n > self.max_steps:
            return StratifiedDecision(False, "rejected_quality", triple_rate, f"step_count={n}")
        if answer_score < self.min_answer_score:
            return StratifiedDecision(False, "rejected_quality", triple_rate, f"answer_score={answer_score:.2f}")

        bucket = self._bucket_for(triple_rate)

        # quota check for the non-rich buckets (rich always accepted)
        total = self.total_accepted
        if bucket == "kg_sparse" and total > 0:
            if self._counts["kg_sparse"] >= self.sparse_quota * (total + 1):
                return StratifiedDecision(False, "kg_sparse", triple_rate, "sparse_quota_full")
        elif bucket == "kg_medium" and total > 0:
            if self._counts["kg_medium"] >= self.medium_quota * (total + 1):
                # demote to acceptance only if it would not break quota; else reject
                return StratifiedDecision(False, "kg_medium", triple_rate, "medium_quota_full")

        self._counts[bucket] += 1
        return StratifiedDecision(True, bucket, triple_rate, "ok")

    def stats(self) -> Dict[str, int]:
        return dict(self._counts)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

@dataclass
class Phase1Config:
    dataset_name: str
    items: Sequence[Dict[str, Any]]
    output_path: str
    teacher_client: TeacherClient
    retriever_factory: Any  # callable returning a retriever, or pre-built object
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


def _annotate_steps(
    raw_output: str,
    kg_subgraph,
    annotator: PRMAnnotator,
) -> List[SilverStepRecord]:
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


def _needs_format_retry(
    steps: List[SilverStepRecord],
    kg_subgraph: Sequence[tuple | list],
    min_steps: int,
) -> bool:
    """Heuristic: retry once when format quality is clearly below Phase1 expectations."""
    if len(steps) < min_steps:
        return True
    if kg_subgraph and not any(step.cited_triples for step in steps):
        return True
    return False


def _build_retry_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    retry_hint = (
        "Regenerate strictly with the required schema. "
        "You MUST output 3-7 steps. "
        "If Knowledge Graph is non-empty, each step must cite at least one triple in "
        "`Knowledge Used: [(head, relation, tail), ...]` and these triples must appear in the KG block. "
        "Do not use `Knowledge Used: []` unless KG is empty. "
        "Keep [Final Answer] concise."
    )
    return [*messages, {"role": "user", "content": retry_hint}]


def _process_one(
    item: Dict[str, Any],
    cfg: Phase1Config,
    retriever_call: _RetrievalAdapter,
) -> Optional["_Candidate"]:
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

    # ---- robust mentions: NER/regex + passage titles ---------------------
    mentions = extract_mentions_robust(question, passages=passages, max_n=8)
    linked = cfg.entity_linker.link(mentions) if mentions else {}
    qids = [q for q in linked.values() if q]

    # ---- SPARQL degrade: empty subgraph is allowed -----------------------
    triples: List = []
    if qids:
        try:
            triples = cfg.kg_retriever.fetch(qids)
        except Exception as exc:  # noqa: BLE001
            logger.warning("SPARQL fetch failed for qid=%s (%d qids): %s", qid, len(qids), exc)
            triples = []

    # coverage is now a *soft* signal only (recorded, never rejects)
    coverage = coverage_score(mentions, linked) if mentions else 0.0

    # Build teacher prompt
    messages = build_teacher_messages(
        question=question,
        retrieved_passages=passages,
        kg_triples=_select_relevant_triples(
            question=question,
            passages=passages,
            triples=triples,
            top_n=cfg.max_kg_triples,
        ),
        top_k=cfg.top_k,
        max_kg_triples=cfg.max_kg_triples,
    )

    raw_output = cfg.teacher_client.chat(messages)
    if not raw_output.strip():
        return None
    annotator = cfg.prm_annotator or PRMAnnotator(entity_linker=cfg.entity_linker, verbose=False)
    parsed_steps = _annotate_steps(raw_output, triples, annotator)

    # One-shot corrective retry when the model ignores required structure.
    min_steps = cfg.accept_filter.min_steps
    if _needs_format_retry(parsed_steps, triples, min_steps):
        retry_messages = _build_retry_messages(messages)
        retry_output = cfg.teacher_client.chat(retry_messages)
        if retry_output.strip():
            retry_steps = _annotate_steps(retry_output, triples, annotator)
            if not _needs_format_retry(retry_steps, triples, min_steps):
                raw_output = retry_output
                parsed_steps = retry_steps

    final_answer = extract_final_answer(raw_output) or ""
    # ---- lenient answer match --------------------------------------------
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
# Per-item candidate + serialised accept/write step. Generation can be
# parallel; the stratified accept decision and the write are serialised so the
# shared quota counter is race-free.
# ---------------------------------------------------------------------------

@dataclass
class _Candidate:
    trajectory: SilverTrajectory
    coverage: float
    answer_score: float


def _decide_and_write(cand: "_Candidate", cfg: Phase1Config, fh) -> Dict[str, Any]:
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


def run_phase1(cfg: Phase1Config) -> Dict[str, Any]:
    """Generate silver trajectories for ``cfg.items`` and write to ``cfg.output_path``.

    Returns a small stats dict.
    """
    out_path = Path(cfg.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build a single retriever object once (heavy).
    retriever = cfg.retriever_factory() if callable(cfg.retriever_factory) else cfg.retriever_factory
    retrieval_call = _RetrievalAdapter(retriever, top_k=cfg.top_k)

    accepted = 0
    total = 0
    written = 0
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
            "phase": "phase1_distill",
            "dataset": cfg.dataset_name,
            "teacher_model": cfg.teacher_client.model,
            "teacher_temperature": cfg.teacher_temperature,
            "seed": cfg.seed,
            "output_path": str(out_path),
            "total_attempts": total,
            "written": written,
            "accepted": accepted,
            "bucket_counts": bucket_counts,
            "accepted_bucket_counts": (
                cfg.accept_filter.stats() if hasattr(cfg.accept_filter, "stats") else {}
            ),
            "accept_filter": vars(cfg.accept_filter),
        },
    )
    logger.info(
        "Phase 1 finished: wrote %d trajectories (accepted=%d / total=%d) buckets=%s to %s",
        written,
        accepted,
        total,
        bucket_counts,
        out_path,
    )
    return {
        "total": total,
        "written": written,
        "accepted": accepted,
        "bucket_counts": bucket_counts,
        "output": str(out_path),
    }


# ---------------------------------------------------------------------------
# Optional helper: build everything from a YAML
# ---------------------------------------------------------------------------

def build_components_from_config(cfg) -> Dict[str, Any]:
    """Convenience builder used by the CLI wrapper.

    Returns a dict with ``entity_linker``, ``kg_retriever``, ``annotator``.
    Caller still supplies items and retriever_factory.
    """
    from kgproweight.retrieval.bootstrap import resolve_entity_cache_path, resolve_kg_cache_dir

    entity_cache = cfg.get("entity_cache_path") or resolve_entity_cache_path()
    kg_cache_dir = cfg.get("kg_cache_dir") or resolve_kg_cache_dir()

    linker = EntityLinker(cache_path=entity_cache)
    retriever = WikidataSubgraphRetriever(
        max_hops=cfg.get("kg_max_hops", 2),
        max_neighbors=cfg.get("kg_max_neighbors", 30),
        cache_dir=kg_cache_dir,
    )
    annotator = PRMAnnotator(entity_linker=linker, verbose=False)
    return {
        "entity_linker": linker,
        "kg_retriever": retriever,
        "annotator": annotator,
        "data_dir": str(data_dir()),
        "index_dir": str(index_dir()),
    }
