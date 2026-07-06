#!/usr/bin/env python
"""Phase 1 CLI (try variant) — generate silver trajectories, improved.

Run from the project root so the local ``try`` modules are importable, e.g.::

    python scripts/train/try/phase1_generate_silver_try.py \
        --config configs/training/phase1_silver_try.yaml \
        --dataset hotpotqa --split train

Improvements over ``scripts/train/phase1_generate_silver.py``:
  * lenient answer matching + stratified KG-density acceptance (see
    ``distill_helpers_try`` / ``phase1_distill_try``),
  * robust mention extraction with passage-title anchors,
  * SPARQL failures degrade to an empty subgraph instead of dropping the item,
  * a **prewarm check** (change 4): the script refuses to start on a cold
    Wikidata cache unless ``--allow_cold_cache`` is passed, nudging you to run
    ``scripts/prepare/04_prewarm_wikidata_cache.py`` first.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Make the sibling try-modules importable regardless of CWD.
# Make sibling try-modules importable regardless of CWD/subdir layout.
_TRY_ROOT = Path(__file__).resolve().parent.parent
for _d in (_TRY_ROOT, _TRY_ROOT / "shared", _TRY_ROOT / "phase1_distill",
           _TRY_ROOT / "phase2_prm", _TRY_ROOT / "phase3_sft", _TRY_ROOT / "phase3_ppo"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

from kgproweight.config import ProjectConfig, load_config
from kgproweight.data.flashrag_loader import flashrag_config
from kgproweight.kg.cache import SubgraphCache
from kgproweight.kg.entity_linker import EntityLinker
from kgproweight.kg.wikidata_retriever import WikidataSubgraphRetriever
from kgproweight.retrieval.bootstrap import resolve_entity_cache_path, resolve_kg_cache_dir
from kgproweight.retrieval.hybrid import DEFAULT_TOPK, build_flashrag_config
from kgproweight.utils.flashrag_bootstrap import setup_flashrag
from kgproweight.utils.logging import configure_logging, get_logger
from kgproweight.utils.paths import data_dir
from kgproweight.utils.seed import set_seed

from distill_helpers_try import StratifiedSilverFilter
from phase1_distill_try import Phase1TryConfig, TeacherClient, run_phase1
from prm_annotator_try import ImprovedPRMAnnotator

configure_logging("INFO")
logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--dataset", default="hotpotqa")
    p.add_argument("--split", default="train")
    # These default to None so we can tell "user passed it" from "use config".
    # CLI-explicit values always win over --config.
    p.add_argument("--max_queries", type=int, default=None)
    p.add_argument("--max_workers", type=int, default=None)
    p.add_argument("--teacher", default=None)
    p.add_argument("--teacher_backend", choices=["openai", "deepseek"], default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    # stratified filter knobs (override yaml/defaults)
    p.add_argument("--min_answer_score", type=float, default=None)
    p.add_argument("--sparse_quota", type=float, default=None)
    p.add_argument("--medium_quota", type=float, default=None)
    # change 4: cold-cache guard
    p.add_argument("--allow_cold_cache", action="store_true",
                   help="Skip the prewarm check (NOT recommended; SPARQL rate limits hurt yield).")
    p.add_argument("--min_cached_subgraphs", type=int, default=50,
                   help="Minimum cached 2-hop subgraphs required before running.")
    return p.parse_args()


def _build_retriever(dataset_name: str):
    setup_flashrag()
    from flashrag.utils import get_retriever

    flashrag_cfg = build_flashrag_config(
        dataset_name=dataset_name,
        save_note="phase1_silver_try",
        save_dir=str(data_dir() / "silver_data" / "_runtime"),
        split="train",
    )
    cfg = flashrag_config(flashrag_cfg)
    return get_retriever(cfg)


def _load_items(dataset_name: str, split: str, max_queries: int, resume_path: Path | None = None) -> List[Dict[str, Any]]:
    src = Path(data_dir()) / dataset_name / f"{split}.jsonl"
    if not src.exists():
        raise FileNotFoundError(f"{src} missing — run scripts/prepare/03_download_datasets.py")
    seen_ids: set[str] = set()
    if resume_path and resume_path.exists():
        with open(resume_path, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    obj = json.loads(line)
                    qid = str(obj.get("qid") or obj.get("id") or "")
                    if qid:
                        seen_ids.add(qid)
                except json.JSONDecodeError:
                    continue
        logger.info("Resume: skipping %d already-processed ids", len(seen_ids))

    items: List[Dict[str, Any]] = []
    with open(src, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(obj.get("id", "")) in seen_ids:
                continue
            items.append(obj)
            if len(items) >= max_queries:
                break
    return items


def _check_prewarm(kg_cache_dir: Path, min_subgraphs: int, allow_cold: bool) -> None:
    """Change 4: insist the Wikidata cache is prewarmed before a big run."""
    cache = SubgraphCache(kg_cache_dir / "kg_subgraph_cache.jsonl")
    n = len(cache)
    if n >= min_subgraphs:
        logger.info("Prewarm check OK: %d cached subgraphs in %s", n, kg_cache_dir)
        return
    msg = (
        f"Wikidata subgraph cache looks cold ({n} < {min_subgraphs} entries at "
        f"{kg_cache_dir}). Run:\n"
        f"  python scripts/prepare/04_prewarm_wikidata_cache.py "
        f"--datasets {{dataset}} --split {{split}}\n"
        f"first, or pass --allow_cold_cache to proceed anyway."
    )
    if allow_cold:
        logger.warning("Cold cache (%d entries) but --allow_cold_cache set; continuing.", n)
        return
    raise SystemExit(msg)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # Baseline defaults (used when neither --config nor CLI provides a value).
    teacher_model = "deepseek-chat"
    teacher_backend = "deepseek"
    temperature = 0.3
    max_queries = 25000
    max_workers = 8
    retrieval_top_k = DEFAULT_TOPK
    out_path = args.output or str(Path(data_dir()) / "silver_data" / "silver_trajectories_try.jsonl")
    flt = StratifiedSilverFilter()

    # Layer 1: --config overrides the hard defaults.
    if args.config:
        cfg = load_config(args.config, validate=ProjectConfig)
        silver = cfg.training.silver_data
        teacher_model = silver.teacher_model
        teacher_backend = silver.teacher_backend
        temperature = silver.teacher_temperature
        max_queries = silver.max_queries
        max_workers = silver.max_workers
        retrieval_top_k = silver.retrieval_top_k
        out_path = silver.output_path or out_path
        # Extra (extra="allow") fields read defensively via getattr.
        flt = StratifiedSilverFilter(
            min_steps=silver.min_steps,
            max_steps=silver.max_steps,
            min_answer_score=getattr(silver, "min_answer_score", flt.min_answer_score),
            rich_triple_rate=getattr(silver, "rich_triple_rate", flt.rich_triple_rate),
            medium_triple_rate=getattr(silver, "medium_triple_rate", flt.medium_triple_rate),
            sparse_quota=getattr(silver, "sparse_quota", flt.sparse_quota),
            medium_quota=getattr(silver, "medium_quota", flt.medium_quota),
        )

    # Layer 2: explicit CLI values (non-None) win over everything above.
    if args.teacher is not None:
        teacher_model = args.teacher
    if args.teacher_backend is not None:
        teacher_backend = args.teacher_backend
    if args.temperature is not None:
        temperature = args.temperature
    if args.max_queries is not None:
        max_queries = args.max_queries
    if args.max_workers is not None:
        max_workers = args.max_workers
    if args.output is not None:
        out_path = args.output

    # CLI overrides win over config.
    if args.min_answer_score is not None:
        flt.min_answer_score = args.min_answer_score
    if args.sparse_quota is not None:
        flt.sparse_quota = args.sparse_quota
    if args.medium_quota is not None:
        flt.medium_quota = args.medium_quota

    entity_cache = resolve_entity_cache_path()
    kg_cache_dir = Path(resolve_kg_cache_dir())

    # Change 4: prewarm guard before the heavy run.
    _check_prewarm(kg_cache_dir, args.min_cached_subgraphs, args.allow_cold_cache)

    out_p = Path(out_path)
    items = _load_items(args.dataset, args.split, max_queries, resume_path=out_p if args.resume else None)

    retriever = _build_retriever(args.dataset)
    linker = EntityLinker(cache_path=entity_cache)
    kg_retr = WikidataSubgraphRetriever(max_hops=2, max_neighbors=30, cache_dir=str(kg_cache_dir))
    annotator = ImprovedPRMAnnotator(entity_linker=linker, verbose=False)
    teacher = TeacherClient(model=teacher_model, backend=teacher_backend, temperature=temperature)

    phase_cfg = Phase1TryConfig(
        dataset_name=args.dataset,
        items=items,
        output_path=out_path,
        append_output=args.resume,
        teacher_client=teacher,
        retriever_factory=retriever,
        entity_linker=linker,
        kg_retriever=kg_retr,
        prm_annotator=annotator,
        top_k=retrieval_top_k,
        max_kg_triples=50,
        max_workers=max_workers,
        accept_filter=flt,
        seed=args.seed,
        teacher_temperature=temperature,
    )
    stats = run_phase1(phase_cfg)
    logger.info("Phase 1 (try) stats: %s", stats)


if __name__ == "__main__":
    main()
