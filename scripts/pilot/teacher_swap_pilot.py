"""Teacher-swap pilot: hold retrieval + KG fixed, vary ONLY the teacher model.

Why not just rerun scripts/train/phase1_generate_silver.py with a different
--teacher: that would also change the retrieval. The existing silver was built
against the 71 GB wiki18 index; the local ``indexes/`` symlinks point at the
989-doc smoke corpus, so a local rerun swaps the teacher AND the passages at
once and the halluc_rate / citation-rate deltas become uninterpretable
(AGENTS.md §4: one variable at a time).

So this replays each record's STORED ``retrieved_passages`` and
``kg_subgraph`` through the same ``build_teacher_messages`` /
``parse_steps`` / ``PRMAnnotator`` path ``_process_one`` uses, and only calls a
different model. Pairing is on question text, since silver qids are
``train_<row>`` while hotpotqa train.jsonl carries hash ids.

Measures the two numbers the rebuild decision rests on:
  * halluc_rate      — cited triples absent from the KG block (baseline 14.53%)
  * step citation rate — steps citing >=1 triple (baseline 51.1%)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from kgproweight.data.parsers import extract_final_answer, parse_steps
from kgproweight.data.prompts import build_teacher_messages
from kgproweight.reward.prm_annotator import PRMAnnotator
from kgproweight.training.phase1_distill import (
    StratifiedSilverFilter,
    TeacherClient,
    _build_retry_messages,
    _needs_format_retry,
    _annotate_steps,
    # answer_match_score lives here, not in kgproweight.eval.metrics --
    # importing kgproweight.eval pulls in flashrag via eval/baselines.py.
    answer_match_score,
)
from kgproweight.kg.entity_linker import EntityLinker
from kgproweight.retrieval.bootstrap import resolve_entity_cache_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("pilot")


def norm(t: Any) -> tuple:
    """Canonical triple key. Silver stores triples as JSON lists, the parser
    yields tuples, and Wikidata labels differ in case/whitespace."""
    if isinstance(t, dict):
        parts = [t.get("head", ""), t.get("relation", ""), t.get("tail", "")]
    else:
        parts = list(t)[:3]
    return tuple(str(p).strip().lower() for p in parts)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="deepseek-v4-flash")
    p.add_argument("--backend", default="deepseek", choices=["deepseek", "openai"])
    p.add_argument("--baseline", default="data/silver_data/silver_v1_reannotated.jsonl")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--max_workers", type=int, default=8)
    p.add_argument("--max_tokens", type=int, default=6000,
                   help="deepseek-v4-* are REASONING models: reasoning_content is billed "
                        "against max_tokens and measured up to ~3.4k on these prompts, so "
                        "TeacherClient's 1500 default returns empty content with "
                        "finish_reason=length. 6000 leaves headroom for the trace itself.")
    p.add_argument("--max_kg_triples", type=int, default=50,
                   help="Must match the baseline run (50) or the KG block differs too.")
    p.add_argument("--top_k", type=int, default=10,
                   help="Passages shown; baseline used --rerank 10.")
    p.add_argument("--output", default=None)
    p.add_argument("--timeout", type=float, default=90.0,
                   help="Per-request timeout. TeacherClient leaves the OpenAI SDK at its "
                        "600s default, so one hanging item can stall a worker for 600s x "
                        "(1+max_retries). The first 300-item run lost 81 records that way: "
                        "they never returned, ThreadPoolExecutor.map was still holding them "
                        "when the iterator ended, and the surviving 219 were biased toward "
                        "fast items -- unusable as a paired comparison.")
    p.add_argument("--only_kg", action="store_true", default=True,
                   help="Restrict to records with a non-empty kg_subgraph — "
                        "halluc_rate and citation rate are undefined without one.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    # The scrubbed launchers require DEEPSEEK_API_KEY; .env only carries the
    # same DeepSeek key under the name OPENAI_API_KEY. Bridge it rather than
    # duplicating a secret into a second variable.
    if args.backend == "deepseek" and not os.environ.get("DEEPSEEK_API_KEY"):
        if os.environ.get("OPENAI_API_KEY"):
            os.environ["DEEPSEEK_API_KEY"] = os.environ["OPENAI_API_KEY"]
            logger.info("DEEPSEEK_API_KEY bridged from OPENAI_API_KEY (same DeepSeek key).")
        else:
            raise SystemExit("no DEEPSEEK_API_KEY / OPENAI_API_KEY in env — source .env first")

    base = [json.loads(l) for l in open(args.baseline, encoding="utf-8") if l.strip()]
    pool = [r for r in base if r.get("kg_subgraph")] if args.only_kg else base
    logger.info("baseline %d records, %d usable (non-empty KG)", len(base), len(pool))
    sample = random.sample(pool, min(args.n, len(pool)))

    out_p = pathlib.Path(args.output or
                         f"data/silver_data/_pilot_{args.teacher.replace('/', '_')}_n{len(sample)}.jsonl")
    out_p.parent.mkdir(parents=True, exist_ok=True)

    teacher = TeacherClient(model=args.teacher, backend=args.backend,
                            temperature=args.temperature, max_tokens=args.max_tokens)
    # Bound every request. Without this a slow item hangs for 600s x 4.
    teacher._client = teacher._client.with_options(timeout=args.timeout)
    linker = EntityLinker(cache_path=resolve_entity_cache_path(), offline=True)
    annotator = PRMAnnotator(entity_linker=linker, verbose=False)
    accept = StratifiedSilverFilter()

    def run_one(rec: Dict[str, Any]) -> Dict[str, Any] | None:
        kg = [tuple(t) if not isinstance(t, dict) else t for t in rec["kg_subgraph"]]
        messages = build_teacher_messages(
            question=rec["question"],
            retrieved_passages=rec.get("retrieved_passages") or [],
            kg_triples=kg,
            top_k=args.top_k,
            max_kg_triples=args.max_kg_triples,
        )
        try:
            raw = teacher.chat(messages)
        except Exception as exc:  # noqa: BLE001
            logger.warning("qid=%s teacher FAILED: %s: %s",
                           rec.get("qid"), type(exc).__name__, str(exc)[:120])
            return {"qid": rec.get("qid"), "_status": "error",
                    "_error": f"{type(exc).__name__}: {str(exc)[:200]}"}
        if not raw.strip():
            logger.warning("qid=%s EMPTY content (reasoning budget?)", rec.get("qid"))
            return {"qid": rec.get("qid"), "_status": "empty"}
        steps = _annotate_steps(raw, kg, annotator)
        retried = False
        if _needs_format_retry(steps, kg, accept.min_steps):
            retried = True
            try:
                raw2 = teacher.chat(_build_retry_messages(messages))
            except Exception:  # noqa: BLE001
                raw2 = ""
            if raw2.strip():
                s2 = _annotate_steps(raw2, kg, annotator)
                if not _needs_format_retry(s2, kg, accept.min_steps):
                    raw, steps = raw2, s2
        gold = str((rec.get("metadata") or {}).get("gold_answer") or "")
        final = extract_final_answer(raw) or ""
        return {
            "_status": "ok",
            "qid": rec.get("qid"),
            "question": rec["question"],
            "answer": final,
            "dataset": rec.get("dataset"),
            "teacher_model": args.teacher,
            "kg_subgraph": [list(t) if not isinstance(t, dict) else t for t in kg],
            "steps": [{"index": s.index, "text": s.text, "label": s.label,
                       "cited_triples": [list(t) for t in s.cited_triples]} for s in steps],
            "teacher_output": raw,
            "retried": retried,
            "metadata": {"gold_answer": gold,
                         "answer_score": answer_match_score(final, gold) if gold else 0.0},
        }

    # as_completed, not map: map yields in submission order, so one slow item
    # blocks reporting and (as in the first run) unfinished futures are simply
    # abandoned when the iterator ends.
    results: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool_ex:
        futs = {pool_ex.submit(run_one, r): r for r in sample}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                r = fut.result()
            except Exception as exc:  # noqa: BLE001
                rec = futs[fut]
                logger.warning("qid=%s worker raised %s", rec.get("qid"), type(exc).__name__)
                r = {"qid": rec.get("qid"), "_status": "worker_error",
                     "_error": f"{type(exc).__name__}: {str(exc)[:200]}"}
            if r is None:
                r = {"qid": None, "_status": "none"}
            (results if r.get("_status") == "ok" else failures).append(r)
            if i % 25 == 0:
                logger.info("  ..%d/%d (ok=%d fail=%d)", i, len(sample), len(results), len(failures))

    with out_p.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Every input item must be accounted for, or the surviving subset is biased
    # toward whatever the model happened to answer quickly.
    n_ok, n_bad = len(results), len(failures)
    logger.info("wrote %d records → %s", n_ok, out_p)
    logger.info("ACCOUNTING: %d submitted = %d ok + %d failed", len(sample), n_ok, n_bad)
    if n_bad:
        by: Dict[str, int] = {}
        for f in failures:
            by[f.get("_status", "?")] = by.get(f.get("_status", "?"), 0) + 1
        logger.warning("failure breakdown: %s", by)
        fp = out_p.with_suffix(".failures.jsonl")
        with fp.open("w", encoding="utf-8") as fh:
            for f in failures:
                fh.write(json.dumps(f, ensure_ascii=False) + "\n")
        logger.warning("failures → %s", fp)
        logger.warning(
            "%.1f%% of items failed. The written subset is NOT a random sample of the "
            "requested %d (slow/hard items drop out first), so paired metrics computed on "
            "it are biased. Raise --timeout or lower --max_workers before comparing.",
            100.0 * n_bad / len(sample), len(sample))
    assert n_ok + n_bad == len(sample), "item accounting lost records"


if __name__ == "__main__":
    main()
