#!/usr/bin/env python
"""Phase 1 CLI — generate silver trajectories.

Wraps :func:`kgproweight.training.phase1_distill.run_phase1`. The script
ensures the Teacher receives *real* RRF top-50 retrieved passages
(bug-fix #4) by instantiating a FlashRAG hybrid retriever before
calling the Teacher.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from kgproweight.config import ProjectConfig, load_config
from kgproweight.data.flashrag_loader import flashrag_config
from kgproweight.kg.entity_linker import EntityLinker
from kgproweight.kg.wikidata_retriever import WikidataSubgraphRetriever
from kgproweight.reward.prm_annotator import PRMAnnotator
from kgproweight.retrieval.bootstrap import resolve_entity_cache_path, resolve_kg_cache_dir
from kgproweight.retrieval.hybrid import DEFAULT_TOPK, build_flashrag_config
from kgproweight.training.phase1_distill import (
    Phase1Config,
    StratifiedSilverFilter,
    TeacherClient,
    run_phase1,
)
from kgproweight.utils.flashrag_bootstrap import setup_flashrag
from kgproweight.utils.logging import configure_logging, get_logger
from kgproweight.utils.paths import data_dir
from kgproweight.utils.seed import set_seed

configure_logging("INFO")
logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None, help="YAML config (see configs/training/phase1_silver.yaml).")
    p.add_argument("--dataset", default="hotpotqa")
    p.add_argument("--split", default="train")
    p.add_argument("--max_queries", type=int, default=25000)
    p.add_argument("--max_workers", type=int, default=8)
    p.add_argument("--teacher", default="deepseek-v4-flash")
    p.add_argument("--teacher_backend", choices=["openai", "deepseek"], default="deepseek")
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--output", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true", help="Skip ids already present in the output file.")
    p.add_argument(
        "--offline",
        choices=["auto", "on", "off"],
        default="auto",
        help="KG/entity lookups: 'on'=cache-only (no Wikidata, misses return empty instantly); "
        "'off'=always call Wikidata; 'auto'=probe once and pick (default).",
    )
    return p.parse_args()


def _build_retriever(dataset_name: str, topk: int = DEFAULT_TOPK):
    """Build a FlashRAG hybrid retriever (RRF top-K)."""
    setup_flashrag()
    from flashrag.utils import get_retriever

    flashrag_cfg = build_flashrag_config(
        dataset_name=dataset_name,
        save_note="phase1_silver",
        save_dir=str(data_dir() / "silver_data" / "_runtime"),
        split="train",
        topk=topk,
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
            qid = str(obj.get("id", ""))
            if qid in seen_ids:
                continue
            items.append(obj)
            if len(items) >= max_queries:
                break
    return items


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.config:
        cfg = load_config(args.config, validate=ProjectConfig)
        silver = cfg.training.silver_data
        teacher_model = silver.teacher_model
        teacher_backend = silver.teacher_backend
        temperature = silver.teacher_temperature
        max_queries = silver.max_queries
        max_workers = silver.max_workers
        out_path = silver.output_path or str(Path(data_dir()) / "silver_data" / "silver_trajectories.jsonl")
        # Stratified acceptance: bucket by KG density with quotas instead of
        # hard-rejecting low-coverage/low-triple-rate traces, so the α-gate sees
        # genuine α→0 fallback examples. Only step bounds map from the legacy
        # config; coverage/triple_rate are now soft (recorded, never reject) and
        # the answer gate uses lenient answer_match_score (default min 0.3).
        accept = StratifiedSilverFilter(
            min_steps=silver.min_steps,
            max_steps=silver.max_steps,
        )
        retrieval_top_k = silver.retrieval_top_k
    else:
        teacher_model = args.teacher
        teacher_backend = args.teacher_backend
        temperature = args.temperature
        max_queries = args.max_queries
        max_workers = args.max_workers
        out_path = args.output or str(Path(data_dir()) / "silver_data" / "silver_trajectories.jsonl")
        accept = StratifiedSilverFilter()
        retrieval_top_k = DEFAULT_TOPK

    out_p = Path(out_path)
    if args.resume:
        items = _load_items(args.dataset, args.split, max_queries, resume_path=out_p)
    else:
        items = _load_items(args.dataset, args.split, max_queries)

    retriever = _build_retriever(args.dataset, topk=retrieval_top_k)

    entity_cache = resolve_entity_cache_path()
    kg_cache_dir = resolve_kg_cache_dir()

    # Decide offline mode. In offline mode, cache misses return empty INSTANTLY
    # instead of blocking on a 10s (entity) / 90s (SPARQL) network timeout each
    # — which would otherwise stretch a 25k run to many hours of pure waiting
    # when Wikidata is unreachable.
    offline = args.offline == "on"
    if args.offline == "auto":
        import requests as _rq

        try:
            _rq.get(
                "https://www.wikidata.org/w/api.php",
                params={"action": "wbsearchentities", "search": "Berlin", "language": "en", "format": "json"},
                headers={"User-Agent": "kgproweight-probe"},
                timeout=8,
            ).raise_for_status()
            offline = False
            logger.info("Wikidata reachable — running ONLINE (cache misses will be fetched live).")
        except Exception as exc:  # noqa: BLE001
            offline = True
            logger.warning(
                "Wikidata UNREACHABLE (%s) — running OFFLINE/cache-only. Cache misses → empty KG "
                "(no per-miss timeout). Entity cache=%s, KG cache=%s.",
                exc, entity_cache, kg_cache_dir,
            )
    else:
        logger.info("Offline mode forced %s by --offline=%s.", "ON" if offline else "OFF", args.offline)

    linker = EntityLinker(cache_path=entity_cache, offline=offline)
    kg_retr = WikidataSubgraphRetriever(max_hops=2, max_neighbors=30, cache_dir=kg_cache_dir, offline=offline)
    annotator = PRMAnnotator(entity_linker=linker, verbose=False)

    teacher = TeacherClient(
        model=teacher_model,
        backend=teacher_backend,
        temperature=temperature,
    )

    phase_cfg = Phase1Config(
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
        accept_filter=accept,
        seed=args.seed,
        teacher_temperature=temperature,
    )
    stats = run_phase1(phase_cfg)
    logger.info("Phase 1 stats: %s", stats)


if __name__ == "__main__":
    main()
