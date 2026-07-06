"""KG-ProWeight inference pipeline.

Uses the canonical SFT/inference schema from :mod:`kgproweight.data.prompts`
(``[Step N] ... [Final Answer]``) instead of FlashRAG's
:class:`ReasoningPipeline` ``<answer>`` protocol.

Workflow per sample:
  1. Hybrid RRF retrieval (top-K passages, configured in FlashRAG config).
  2. Optional Wikidata 2-hop subgraph injection (honours D_dropout overrides).
  3. Single-pass generation via :func:`build_inference_messages`.
  4. Answer extraction via :func:`extract_final_answer`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from kgproweight.data.parsers import parse_steps
from kgproweight.data.prompts import build_inference_messages
from kgproweight.eval.pred_processing import extract_kg_proweight_answer
from kgproweight.kg.entity_linker import EntityLinker, extract_mentions
from kgproweight.kg.wikidata_retriever import WikidataSubgraphRetriever
from kgproweight.retrieval.hybrid import DEFAULT_TOPK
from kgproweight.reward.alpha_gate import AlphaGate, compute_features
from kgproweight.reward.prm_annotator import PRMAnnotator
from kgproweight.utils.flashrag_bootstrap import setup_flashrag
from kgproweight.utils.logging import get_logger

setup_flashrag()

from flashrag.pipeline.pipeline import BasicPipeline  # noqa: E402
from flashrag.utils import get_generator, get_retriever  # noqa: E402

logger = get_logger(__name__)


class KGProWeightPipeline(BasicPipeline):
    """Single-pass inference with KG context + α telemetry."""

    def __init__(
        self,
        config,
        alpha_gate_path: Optional[str] = None,
        entity_cache_path: Optional[str] = None,
        kg_cache_dir: Optional[str] = None,
        record_alpha: bool = True,
        inject_kg: bool = True,
        max_kg_triples: int = 50,
        max_mentions: int = 5,
        retrieval_topk: Optional[int] = None,
        generator=None,
        retriever=None,
        **kwargs,
    ) -> None:
        super().__init__(config, prompt_template=kwargs.pop("prompt_template", None))
        self.record_alpha = record_alpha
        self.inject_kg = inject_kg
        self.max_kg_triples = max_kg_triples
        self.max_mentions = max_mentions
        _cfg_topk = config["retrieval_topk"] if "retrieval_topk" in config else DEFAULT_TOPK
        self.retrieval_topk = retrieval_topk or int(_cfg_topk)
        self._alpha_records: List[Dict] = []

        self.generator = generator if generator is not None else get_generator(config)
        self.retriever = retriever if retriever is not None else get_retriever(config)

        # KGPW_KG_OFFLINE=1 makes entity linking + subgraph fetch never touch the
        # network: cache hits still work, misses return empty INSTANTLY (no 10-100s
        # SPARQL/Search timeouts). Required when Wikidata is unreachable (e.g. CN host).
        _kg_offline = os.environ.get("KGPW_KG_OFFLINE", "").lower() in ("1", "true", "yes")
        if _kg_offline:
            logger.info("KG offline mode ON (KGPW_KG_OFFLINE) — cache-only, no network.")
        self.entity_linker = EntityLinker(cache_path=entity_cache_path, offline=_kg_offline)
        self.kg_retriever = WikidataSubgraphRetriever(
            max_hops=2, max_neighbors=30, cache_dir=kg_cache_dir, offline=_kg_offline
        )
        self.prm_annotator = PRMAnnotator(entity_linker=self.entity_linker, verbose=False)

        self.alpha_gate = AlphaGate()
        if alpha_gate_path and Path(alpha_gate_path).exists():
            self.alpha_gate.load_state_dict(torch.load(alpha_gate_path, map_location="cpu"))
            logger.info("Loaded AlphaGate from %s", alpha_gate_path)
        elif inject_kg:
            logger.warning("No AlphaGate checkpoint provided — using initial weights.")
        self.alpha_gate.eval()

    # ------------------------------------------------------------------
    # KG context construction
    # ------------------------------------------------------------------

    def _get_dropout_kg(self, item) -> Optional[List[Tuple[str, str, str]]]:
        """Return the severed subgraph if this item is from D_dropout."""
        meta = getattr(item, "metadata", None) or {}
        if isinstance(meta, dict):
            dropout = meta.get("dropout")
            if isinstance(dropout, dict):
                mod = dropout.get("modified_kg")
                if isinstance(mod, list) and mod:
                    return [tuple(t) for t in mod if len(t) == 3]
        return None

    def _build_kg_context(self, item) -> List[Tuple[str, str, str]]:
        if not self.inject_kg:
            return []

        dropout = self._get_dropout_kg(item)
        if dropout is not None:
            return list(dropout)

        mentions = extract_mentions(item.question, max_n=self.max_mentions)
        linked = self.entity_linker.link(mentions)
        qids = [q for q in linked.values() if q]
        return self.kg_retriever.fetch(qids) if qids else []

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, dataset, do_eval: bool = True, pred_process_fun=None):
        questions = list(dataset.question)
        logger.info("KGProWeight inference on %d samples (top_k=%d, inject_kg=%s)",
                    len(questions), self.retrieval_topk, self.inject_kg)

        retrieval_results = self.retriever.batch_search(questions)
        dataset.update_output("retrieval_result", retrieval_results)

        prompts: List[str] = []
        kg_subgraphs: List[List[Tuple[str, str, str]]] = []
        used_dropout: List[bool] = []

        for item, passages in zip(dataset, retrieval_results):
            kg_sub = self._build_kg_context(item)
            kg_subgraphs.append(kg_sub)
            used_dropout.append(self._get_dropout_kg(item) is not None)
            msgs = build_inference_messages(
                question=item.question,
                retrieved_passages=passages,
                kg_triples=kg_sub,
                top_k=self.retrieval_topk,
                max_kg_triples=self.max_kg_triples,
            )
            prompts.append(self.prompt_template.get_string(messages=msgs))

        dataset.update_output("prompt", prompts)
        dataset.update_output("kg_subgraphs", kg_subgraphs)
        dataset.update_output("used_dropout_kg", used_dropout)

        raw_outputs = self.generator.generate(prompts)
        dataset.update_output("raw_output", raw_outputs)

        preds: List[str] = []
        alpha_stats: List[Dict] = []
        ihr_stats: List[Dict] = []

        for item, raw_output, kg_sub in zip(dataset, raw_outputs, kg_subgraphs):
            pred = extract_kg_proweight_answer(raw_output)
            preds.append(pred)
            alpha_stats.append(self._compute_alpha_stats(raw_output, kg_sub, item.question))
            ihr_stats.append(self._compute_ihr(raw_output, kg_sub))

        dataset.update_output("pred", preds)
        dataset.update_output("alpha_stats", alpha_stats)
        dataset.update_output("ihr_flags", ihr_stats)

        if self.record_alpha:
            self._alpha_records.extend(alpha_stats)

        return self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)

    # ------------------------------------------------------------------
    # Telemetry helpers
    # ------------------------------------------------------------------

    def _compute_alpha_stats(self, generated_text: str, kg_subgraph, query: str) -> Dict:
        steps = parse_steps(generated_text) if generated_text else []
        alphas: List[float] = []
        for step in steps:
            f_density, f_confidence, f_entropy = compute_features(
                step_entities=step.mentioned_entities,
                kg_subgraph=kg_subgraph,
                logprobs=None,
                entity_linker=self.entity_linker,
            )
            alphas.append(self.alpha_gate.forward_single(f_density, f_confidence, f_entropy))

        mean_alpha = sum(alphas) / len(alphas) if alphas else 0.0
        var = (sum((a - mean_alpha) ** 2 for a in alphas) / len(alphas)) if len(alphas) > 1 else 0.0
        return {
            "query": query,
            "num_steps": len(steps),
            "alpha_mean": mean_alpha,
            "alpha_std": var ** 0.5,
            "alpha_values": alphas,
        }

    def _compute_ihr(self, generated_text: str, kg_subgraph) -> Dict:
        if not generated_text or not kg_subgraph:
            return {"ihr_heuristic": None, "n_steps": 0, "n_hallucinated": 0}
        steps = parse_steps(generated_text)
        labels = self.prm_annotator.annotate_trajectory(steps, kg_subgraph)
        n_neg = sum(1 for x in labels if x == -1)
        total = len(labels)
        return {
            "ihr_heuristic": (n_neg / total) if total else 0.0,
            "n_steps": total,
            "n_hallucinated": n_neg,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_alpha_distribution(self, output_path: str) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            for record in self._alpha_records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("Saved α distribution to %s", output_path)

    def print_alpha_summary(self) -> None:
        if not self._alpha_records:
            logger.info("No α records collected.")
            return
        all_alphas = [a for r in self._alpha_records for a in r.get("alpha_values", [])]
        if not all_alphas:
            return
        mean_alpha = sum(all_alphas) / len(all_alphas)
        std_alpha = (sum((a - mean_alpha) ** 2 for a in all_alphas) / len(all_alphas)) ** 0.5
        logger.info(
            "α summary: n=%d mean=%.4f std=%.4f min=%.4f max=%.4f",
            len(all_alphas),
            mean_alpha,
            std_alpha,
            min(all_alphas),
            max(all_alphas),
        )
